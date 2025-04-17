import { TestBed } from '@angular/core/testing';

import { ServletService } from './servlet.service';

describe('ServletService', () => {
  let service: ServletService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ServletService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
